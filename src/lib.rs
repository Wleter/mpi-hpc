pub extern crate bincode;
pub extern crate mpi;

#[macro_export]
macro_rules! distribute {
    (|| $setup:expr, 
        |$x:ident: $in_type:ty| $computation:expr, 
        |$xs:ident: Vec<$out_type:ty>| $finalize:expr
    ) => {
        use $crate::mpi::traits::*;

        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let rank = world.rank();
        let size = world.size();

        let points: Vec<$in_type> = if rank == 0 {
            $setup
        } else {
            Vec::new()
        };

        let local_points: Vec<$in_type> = if rank == 0 {
            let total = points.len() as i32;
            let chunk_size = (total + size - 1) / size;

            let own_chunk: Vec<$in_type> = points
                .iter()
                .cloned()
                .take(chunk_size as usize)
                .collect();
    
            for i in 1..size {
                let chunk: Vec<$in_type> = points
                    .iter()
                    .cloned()
                    .skip((i * chunk_size) as usize)
                    .take(chunk_size as usize)
                    .collect();
                world.process_at_rank(i).send(&chunk);
            }
            own_chunk
        } else {
            let (received_chunk, _status) = world.process_at_rank(0).receive_vec::<f64>();
            received_chunk
        };

        let local_results: Vec<$out_type> = local_points.into_iter()
            .map(|$x| {
                $computation
            })
            .collect();

        if rank == 0 {
            let mut $xs = local_results;
            for i in 1..size {
                let (serialized_bytes, _status) = world.process_at_rank(i).receive_vec::<u8>();
                let mut received_results: Vec<$out_type> =
                    $crate::bincode::deserialize(&serialized_bytes)
                        .expect("Deserialization failed");
                $xs.append(&mut received_results);
            }

            $finalize
        } else {
            // For non-root processes, serialize the local results and send them to rank 0.
            let serialized = $crate::bincode::serialize(&local_results)
                .expect("Serialization failed");
            world.process_at_rank(0).send(&serialized);
        }
    };
}